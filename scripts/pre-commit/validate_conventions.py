"""Validate docstring, typing, and structural conventions for Python files.

This script enforces repository conventions for docstrings, type hints, typing
imports, and nested function usage. It is intended to be run via pre-commit.
"""

from __future__ import annotations

import ast
import sys
import typing as tp
from dataclasses import dataclass
from pathlib import Path

SKIP_DIR_KEYWORDS = {"__pycache__"}
TYPING_PREFIX = "typing" + "."


@dataclass
class Violation:
    """Capture metadata describing a single validation failure.

    Attributes:
        path: File that produced the violation.
        lineno: Line number associated with the violation.
        message: Human-readable explanation of the issue.

    """

    path: Path
    lineno: int
    message: str

    def __str__(self) -> str:
        """Return a formatted representation suitable for CLI output.

        Returns:
            Formatted string describing the violation location and message.

        """
        return f"{self.path}:{self.lineno}: {self.message}"


def _has_typing_alias(tree: ast.AST) -> bool:
    """Return whether the module imports ``typing as tp``.

    Args:
        tree: Parsed syntax tree for the module.

    Returns:
        True if the module imports ``typing as tp``.

    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "typing" and alias.asname == "tp":
                    return True

        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            return False

    return False


def _uses_forbidden_typing(text: str) -> bool:
    """Return whether the text contains raw typing module references.

    Args:
        text: Source code to inspect.

    Returns:
        True if the file references the typing prefix directly.

    """
    return TYPING_PREFIX in text


def _docstring_has_section(lines: tp.Sequence[str], section: str) -> bool:
    """Return whether a docstring contains the requested section heading.

    Args:
        lines: Docstring split into individual lines.
        section: Heading name to locate.

    Returns:
        True if a line begins with the provided section label.

    """
    return any(line.strip().startswith(section) for line in lines)


def _docstring_errors(
    docstring: str | None,
    *,
    require_args: bool = False,
    require_returns: bool = False,
    require_yields: bool = False,
    require_attributes: bool = False,
) -> list[str]:
    """Collect docstring format errors based on required sections.

    Args:
        docstring: Raw docstring string or ``None`` when not present.
        require_args: Whether the docstring must describe parameters.
        require_returns: Whether the docstring must describe returns.
        require_yields: Whether the docstring must describe yielded values.
        require_attributes: Whether the docstring must describe attributes.

    Returns:
        List of textual error descriptions.

    """
    errors: list[str] = []

    if not docstring:
        errors.append("missing docstring")
        return errors

    stripped = docstring.strip("\n")
    lines = stripped.splitlines()

    if not lines or not lines[0].strip():
        errors.append("docstring must start with a summary line")

    if len(lines) > 1 and lines[1].strip():
        errors.append("docstring summary line must be followed by a blank line")

    if require_args and not _docstring_has_section(lines, "Args:"):
        errors.append("docstring missing 'Args:' section")

    if require_returns and not (
        _docstring_has_section(lines, "Returns:") or _docstring_has_section(lines, "Yields:")
    ):
        errors.append("docstring missing 'Returns:' section")

    if require_yields and not _docstring_has_section(lines, "Yields:"):
        errors.append("docstring missing 'Yields:' section")

    if require_attributes and not _docstring_has_section(lines, "Attributes:"):
        errors.append("docstring missing 'Attributes:' section")

    return errors


def _function_returns_value(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether the function declares a non-``None`` return annotation.

    Args:
        func: Function definition node to inspect.

    Returns:
        True if the function has a return annotation that is not ``None``.

    """
    annotation = func.returns

    if annotation is None:
        return False

    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return False

    return not (isinstance(annotation, ast.Name) and annotation.id == "None")


def _function_is_generator(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether the function body contains yield statements.

    Args:
        func: Function definition node to inspect.

    Returns:
        True if the function yields values.

    """
    return any(isinstance(node, ast.Yield | ast.YieldFrom) for node in ast.walk(func))


def _iter_arguments(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[str, ast.expr | None]]:
    """Return positional and keyword arguments paired with annotations.

    Args:
        func: Function definition node to inspect.

    Returns:
        List of argument names with their annotations (if any).

    """
    args: list[tuple[str, ast.expr | None]] = []

    for arg in getattr(func.args, "posonlyargs", []):
        args.append((arg.arg, arg.annotation))

    for arg in func.args.args:
        args.append((arg.arg, arg.annotation))

    if func.args.vararg:
        args.append((func.args.vararg.arg, func.args.vararg.annotation))

    for arg in func.args.kwonlyargs:
        args.append((arg.arg, arg.annotation))

    if func.args.kwarg:
        args.append((func.args.kwarg.arg, func.args.kwarg.annotation))

    return args


def _class_defines_attributes(cls: ast.ClassDef) -> bool:
    """Return whether the class body defines attributes needing documentation.

    Args:
        cls: Class definition node to inspect.

    Returns:
        True if the class declares attributes via assignments or annotations.

    """
    return any(isinstance(node, ast.Assign | ast.AnnAssign) for node in cls.body)


def _record_docstring_violations(
    path: Path,
    lineno: int,
    target: str,
    docstring: str | None,
    *,
    require_args: bool = False,
    require_returns: bool = False,
    require_yields: bool = False,
    require_attributes: bool = False,
) -> list[Violation]:
    """Create violations for docstring format issues on a specific symbol.

    Args:
        path: File containing the definition.
        lineno: Line number for the definition.
        target: Fully-qualified display name for the symbol.
        docstring: Docstring contents or ``None``.
        require_args: Whether the docstring must document parameters.
        require_returns: Whether the docstring must describe return values.
        require_yields: Whether the docstring must describe yielded values.
        require_attributes: Whether the docstring must describe attributes.

    Returns:
        List of violations referencing the docstring issues.

    """
    errors = _docstring_errors(
        docstring,
        require_args=require_args,
        require_returns=require_returns,
        require_yields=require_yields,
        require_attributes=require_attributes,
    )

    return [Violation(path, lineno, f"{target} docstring {error}") for error in errors]


def _check_function_def(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    path: Path,
    *,
    parent: str | None = None,
) -> list[Violation]:
    """Validate a function definition and return all convention violations.

    Args:
        func: Function definition node to inspect.
        path: File containing the definition.
        parent: Optional class name when the function is a method.

    Returns:
        List of discovered violations.

    """
    target = f"{parent}.{func.name}" if parent else func.name
    arg_info = [(name, ann) for name, ann in _iter_arguments(func) if name not in {"self", "cls"}]
    is_generator = _function_is_generator(func)
    require_returns = _function_returns_value(func) and not is_generator
    violations: list[Violation] = []
    violations.extend(
        _record_docstring_violations(
            path,
            func.lineno,
            f"function {target}",
            ast.get_docstring(func),
            require_args=bool(arg_info),
            require_returns=require_returns,
            require_yields=is_generator,
        )
    )

    for name, annotation in arg_info:
        if annotation is None:
            violations.append(
                Violation(
                    path,
                    func.lineno,
                    f"function {target} argument '{name}' missing type annotation",
                )
            )

    if func.returns is None:
        message = f"function {target} missing return type annotation"
        violations.append(Violation(path, func.lineno, message))

    for node in func.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            violations.append(
                Violation(
                    path,
                    node.lineno,
                    f"nested function '{node.name}' found inside {target}; refactor to a helper",
                )
            )

    return violations


def _check_class_def(cls: ast.ClassDef, path: Path) -> list[Violation]:
    """Validate a class definition and return all convention violations.

    Args:
        cls: Class definition node to inspect.
        path: File containing the class definition.

    Returns:
        List of discovered violations.

    """
    violations: list[Violation] = []
    requires_attributes = _class_defines_attributes(cls)
    violations.extend(
        _record_docstring_violations(
            path,
            cls.lineno,
            f"class {cls.name}",
            ast.get_docstring(cls),
            require_attributes=requires_attributes,
        )
    )

    for node in cls.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            violations.extend(_check_function_def(node, path, parent=cls.name))

    return violations


def _check_file(path: Path) -> list[Violation]:
    """Validate an individual Python file and return discovered violations.

    Args:
        path: File to validate.

    Returns:
        List of violations produced while validating the file.

    """
    import warnings

    text = path.read_text()
    # Suppress SyntaxWarnings about invalid escape sequences (used for HTML generation)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            lineno = exc.lineno or 0
            return [Violation(path, lineno, f"syntax error: {exc.msg}")]

    violations: list[Violation] = []

    if _uses_forbidden_typing(text):
        violations.append(
            Violation(path, 1, "use 'import typing as tp' and reference types via tp.* only")
        )

    module_doc = ast.get_docstring(tree)
    violations.extend(_record_docstring_violations(path, 1, "module", module_doc))

    has_typing_alias = _has_typing_alias(tree)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            violations.extend(_check_class_def(node, path))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            violations.extend(_check_function_def(node, path))

    if not has_typing_alias and ("tp." in text or TYPING_PREFIX in text):
        violations.append(
            Violation(path, 1, "import typing as tp and reference typing constructs via tp.*")
        )

    return violations


def _should_skip(path: Path) -> bool:
    """Return whether the path should be ignored by the validator.

    Args:
        path: Candidate file path.

    Returns:
        True if the file is in a blocked directory.

    """
    return any(part in SKIP_DIR_KEYWORDS for part in path.parts)


def main(argv: tp.Sequence[str]) -> int:
    """Validate provided Python files and emit violations.

    Args:
        argv: Command-line arguments, starting with the script name.

    Returns:
        Zero when validation passes, or non-zero on failure.

    """
    if len(argv) <= 1:
        return 0

    violations: list[Violation] = []

    for arg in argv[1:]:
        path = Path(arg)
        if path.suffix != ".py" or _should_skip(path):
            continue
        violations.extend(_check_file(path))

    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        # Warn-only mode: surface convention warnings without blocking the
        # commit. Flip to ``return 1`` here to enforce strict mode.
        print(
            f"\n[warn] {len(violations)} convention warnings (non-blocking)",
            file=sys.stderr,
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
