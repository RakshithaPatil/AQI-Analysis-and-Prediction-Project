Set-Location "C:\Users\Rakshitha Patil\air_quality\aqi_prediction"
$env:PYTHONPATH = "C:\Users\Rakshitha Patil\air_quality\aqi_prediction"
& "C:\Users\Rakshitha Patil\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run dashboard/app.py --server.port 8501 --server.headless false
