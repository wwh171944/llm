@echo off
call conda activate llm
cd /d E:\llm_project\viz
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
pause