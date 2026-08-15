/* ============================================================
   Shared quiz + recall widgets for the mtg-agent teaching workspace.
   Zero dependencies. Include with <script src="../assets/quiz.js" defer>.

   MULTIPLE CHOICE
   ---------------
   <div class="quiz" data-answer="1">
     <p class="q"><span class="num">Q1</span>Question text?</p>
     <button class="opt">option a</button>
     <button class="opt">option b</button>   <!-- index 1 = correct -->
     <div class="fb"><p>Explanation shown after answering.</p></div>
   </div>

   RECALL (free response, self-graded — the retrieval that builds storage
   strength; grading yourself is the point, so there is nothing to score)
   ---------------------------------------------------------------------
   <div class="recall">
     <p>Prompt?</p>
     <textarea placeholder="from memory..."></textarea>
     <button class="reveal">Show answer</button>
     <div class="answer"><p>The answer.</p></div>
   </div>
   ============================================================ */

(function () {
  "use strict";

  function initQuiz(quiz) {
    var correct = parseInt(quiz.dataset.answer, 10);
    var opts = Array.prototype.slice.call(quiz.querySelectorAll("button.opt"));
    var fb = quiz.querySelector(".fb");

    opts.forEach(function (btn, i) {
      btn.addEventListener("click", function () {
        if (quiz.dataset.done) return;
        quiz.dataset.done = "1";

        opts.forEach(function (b) { b.disabled = true; });
        opts[correct].classList.add("right");
        if (i !== correct) btn.classList.add("wrong");
        if (fb) fb.classList.add("show");

        document.dispatchEvent(new CustomEvent("quiz:answered", {
          detail: { correct: i === correct }
        }));
      });
    });
  }

  function initRecall(recall) {
    var btn = recall.querySelector(".reveal");
    var ans = recall.querySelector(".answer");
    if (!btn || !ans) return;
    btn.addEventListener("click", function () {
      ans.classList.toggle("show");
      btn.textContent = ans.classList.contains("show") ? "Hide answer" : "Show answer";
    });
  }

  function initScore(el, total) {
    var right = 0, done = 0;
    function render() {
      el.textContent = done === 0
        ? total + " questions"
        : right + " / " + done + " correct" + (done === total ? " — done" : "");
    }
    document.addEventListener("quiz:answered", function (e) {
      done++;
      if (e.detail.correct) right++;
      render();
    });
    render();
  }

  document.addEventListener("DOMContentLoaded", function () {
    var quizzes = document.querySelectorAll(".quiz[data-answer]");
    quizzes.forEach(initQuiz);
    document.querySelectorAll(".recall").forEach(initRecall);
    var score = document.querySelector(".score");
    if (score) initScore(score, quizzes.length);
  });
})();
