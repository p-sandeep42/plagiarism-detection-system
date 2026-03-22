$backend = Start-Process -NoNewWindow -FilePath "powershell" -ArgumentList "-Command cd api; .\venv\Scripts\activate; uvicorn main:app --reload --port 8000" -PassThru
$frontend = Start-Process -NoNewWindow -FilePath "powershell" -ArgumentList "-Command npm run dev" -PassThru

Write-Host "AuraDiff is running!"
Write-Host "Backend API: http://localhost:8000"
Write-Host "Frontend App: http://localhost:3000"
Write-Host "Press any key to stop both servers..."

$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Stop-Process -Id $backend.Id -Force
Stop-Process -Id $frontend.Id -Force
Write-Host "Stopped AuraDiff."
