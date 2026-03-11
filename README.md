# 實時面部情緒識別系統 (AI Emotion Recognition)

## 專案簡介
本專案為一個基於 Flask 框架的 Web AI 應用，能夠透過網頁瀏覽器開啟攝像頭，即時捕捉並辨識使用者的七種基本情緒（生氣、厭惡、害怕、開心、面無表情、難過與驚訝）。

## 團隊與個人貢獻
本專案為大學四年級之團隊專題作品，整體模型訓練在 70:30 的資料分割下取得 73.1% 的準確率。
**我在本專案中的主要貢獻：**
* **後端架構建立**：獨立使用 Python Flask 建立串流伺服器，實作即時影像擷取 (`/video_feed`) 與前端 API 溝通邏輯。
* **環境建置與系統優化**：成功克服 Windows 系統下 `dlib` 與 C++ 編譯器的相容性問題，並利用 Conda 建立隔離的虛擬環境以確保系統穩定運行。
* **AI 模型工程化整合**：負責將團隊訓練好的 `scikit-learn` 集成學習模型 (`.joblib`) 與 Dlib 68 點地標預測器匯入後端應用，實作特徵擷取函數 (`compute_features`) 並輸出預測結果。
*(註：為尊重團隊智慧財產權，本開源庫僅展示本人負責之 Web 應用與模型介接實作，未附上完整學術報告與海報。)*

## 開發技術
* **後端與 API**：Python 3.12, Flask
* **電腦視覺**：OpenCV, Dlib (68-face landmarks)
* **機器學習**：Scikit-learn, Joblib (SVM & Random Forest 集成模型)
* **特徵工程**：Local Binary Pattern (LBP), 歐幾里得幾何距離計算

## 如何在本地端執行
1. 安裝環境依賴：
   `pip install flask opencv-python dlib scikit-learn numpy skimage joblib`
2. 下載 AI 模型：
   請至 [https://drive.google.com/drive/folders/1Pif6jQ-5dXMbkps-7zRLZChyWUFNVhZA?usp=drive_link] 下載 `ensemble_model.joblib` 與 `shape_predictor_68_face_landmarks.dat`，並放置於專案根目錄中。
3. 啟動伺服器：
   `python app.py`
4. 開啟瀏覽器進入 `http://127.0.0.1:5000` 即可使用。
