const riskForm = document.getElementById("riskForm");
const riskLevel = document.getElementById("riskLevel");
const riskNote = document.getElementById("riskNote");

const getRisk = ({ attendance, marks, delay }) => {
  let score = 0;

  if (attendance < 70) score += 2;
  if (attendance < 80) score += 1;

  if (marks < 50) score += 2;
  if (marks < 65) score += 1;

  if (delay > 10) score += 2;
  if (delay > 5) score += 1;

  if (score >= 5) return "High";
  if (score >= 3) return "Medium";
  return "Low";
};

const updateDisplay = (risk, summary) => {
  riskLevel.textContent = risk;
  riskNote.textContent = summary;
};

riskForm.addEventListener("submit", (event) => {
  event.preventDefault();

  const attendance = Number(document.getElementById("attendance").value);
  const marks = Number(document.getElementById("marks").value);
  const delay = Number(document.getElementById("delay").value);

  const risk = getRisk({ attendance, marks, delay });
  const summary = `Attendance: ${attendance}%, Marks: ${marks}, Delay: ${delay} days.`;

  updateDisplay(risk, summary);
});
