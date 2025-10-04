document.getElementById('predict').addEventListener('click', async () => {
  const payload = {
    'Orbital Period Days': parseFloat(document.getElementById('orbital_period').value) || null,
    'Orbit Semi-Major Axis': parseFloat(document.getElementById('semi_major').value) || null,
    'Eccentricity': parseFloat(document.getElementById('eccentricity').value) || null,
    'Stellar Mass': parseFloat(document.getElementById('stellar_mass').value) || null,
    'Discovery Method': document.getElementById('discovery_method').value,
    'Spectral Type': document.getElementById('spectral_type').value || null
  };

  const resultEl = document.getElementById('result');
  resultEl.textContent = 'Predicting...';

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await resp.json();
    if (resp.ok) {
      resultEl.textContent = 'Predicted Mass: ' + data.prediction;
    } else {
      resultEl.textContent = 'Error: ' + (data.error || JSON.stringify(data));
    }
  } catch (err) {
    resultEl.textContent = 'Request failed: ' + err;
  }
});
