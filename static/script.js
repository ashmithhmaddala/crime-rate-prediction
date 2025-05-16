document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("prediction-form");
  const resultBox = document.getElementById("result-box");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Get form values
    const temperature = parseFloat(document.getElementById("temperature").value);
    const rainfall = parseFloat(document.getElementById("rainfall").value);
    const severity = document.getElementById("severity").value;
    const reported = document.getElementById("reported").value;
    const responseTime = parseFloat(document.getElementById("response_time").value);
    const timeOfDay = document.getElementById("time_of_day").value;
    const socioZone = document.getElementById("socio_zone").value;
    const area = document.getElementById("area").value;

    // Validation
    if (
      isNaN(temperature) || isNaN(rainfall) || isNaN(responseTime) ||
      !severity || !reported || !timeOfDay || !socioZone || !area
    ) {
      resultBox.innerHTML = `<p style="color:#ff4c24;">Please fill in all fields correctly.</p>`;
      return;
    }

    // Build payload
    const payload = {
      temperature,
      rainfall,
      severity,
      reported,
      response_time: responseTime,
      time_of_day: timeOfDay,
      socio_zone: socioZone,
      area
    };

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (response.ok) {
        resultBox.innerHTML = `
          <p style="color:#0ba954;">
            <strong>Prediction:</strong> ${data.prediction === 1 ? "Crime Likely" : "No Crime Detected"}<br>
            <strong>Probability:</strong> ${data.probability}%
          </p>
        `;
      } else {
        resultBox.innerHTML = `<p style="color:#ff4c24;">Prediction failed. Try again.</p>`;
      }
    } catch (error) {
      console.error("Error:", error);
      resultBox.innerHTML = `<p style="color:#ff4c24;">Server error. Please try later.</p>`;
    }
  });
});
