# 招商案件 RAG 伺服器

這個儲存庫就是招商案件分析服務，與網站 `ken19cars-cmd/ur-map-vercel` 分開。

預設不需要付費 API 金鑰，也不會呼叫 Gemini：嵌入是字元 n-gram，沒接模型時用抽取式填寫。可選的聊天模型只接 OpenAI 相容端點（Ollama 放在另一台、只走 Tailscale），而且只用於 Notion 案件 JSON。

## 兩種案件分開處理

**進行中**的案子來自本機資料夾，不爬 Ken 的磁碟、也不假設雲端看得到 `D:`。

```text
CASES_ROOT/
  1.都市更新/00基隆市/基隆市七堵區一德新村公辦都更案/申請須知.pdf
  2.捷運聯開/...
  3.促參/...
  4.招標標售/...
```

四層是：案件類型 / 縣市（資料夾名前綴數字會去掉，`00基隆市` 存成 `基隆市`）/ 案名 / PDF 或 txt。檔名用來標文件種類：申請須知、需求說明書、評選文件、實施(出資)契約，其餘標成其他。

**招標完成**的案子不從這個資料夾來。預設讀一份 JSON 或 CSV。有設 Notion token 與 database id、且沒有離線檔時，才向 Notion 查 `招商情形 = 招標完成`。公開網頁研究另外打 `/api/completed/research`，只抓你給的網址（或測試注入的搜尋結果），遵守 `robots.txt` 與請求間隔，收下最優申請人、得標、規劃相關文字，以及 alt / 鄰近文字像建築效果模擬圖的圖片網址。

## 它取代 n8n 裡的 Gemini 節點

分析契約在 `prompts/notion-case-analysis.md`。`POST /api/notion/analyze` 只回第 38 節的 JSON：

`pageId`、`caseName`、`baseTable`、`landUseTable`、`updates`。

`pageId` 與 `caseName` 照請求原樣抄回。`updates` 只用這次 `templateBlocks` 裡真實存在的 block id。PDF 沒寫的欄位留空，不補、不推。

對照已填好的案件頁，各段內容長這樣（測試用的是合成的「示範」案，不是任一筆真實案件）：

| 區塊 | 內容形狀 |
| --- | --- |
| ❯計畫概述 | 一段短文 |
| ❯基地基本資料 | `baseTable.bases`：基地別、座落、範圍、面積、土地權屬。權屬只留「所有人＋比例」，不放面積 |
| ❯周邊環境概況 | 短文 |
| ❯使用強度規範 | 分區 + `建蔽率60%、容積率200%`（不寫法定）+ ☑／☒ 容積項目 + 退縮、倍數這類數字條件 |
| ❯意願調查 | 只留同意比例／同意率／不同意比例 |
| ❯開發定位/規劃構想 | 短條列 |
| ❯開發權利義務/負擔條件 | 最多 6 列，每列有標題與壓縮條列 |
| ❯期程規範 | 簽約日、開工日這類起算點與期限。保證金不放這裡 |
| ❯財務條件 | `≥`／`≤` 與保證金金額 |
| ❯申請人資格 | 一般資格列是「單一申請人｜合作聯盟(N家以內)」；其餘列分單一申請人與合作聯盟。PDF 寫「或」就保留「或」 |

空白 Markdown 匯出有 ❯期程規範 標題。某份已填頁的 Markdown 沒有這段文字，多半是時程圖只以圖片存在。分析仍會填期程 paragraph，不會因為那份匯出沒字就拿掉。

## 什麼已經可用、什麼是 Phase 2

| n8n 分支 | 狀態 |
| --- | --- |
| 掃描案件資料夾、抽出並合併 PDF 文字、分析成 Notion JSON | 可用 |
| 本機提問、抽取式欄位（產業、地點、金額、時程、聯絡窗口） | 可用，不走聊天模型 |
| 招標完成清單（JSON/CSV，或 Notion 查詢）與公開頁研究 | 可用。研究要自備網址，不會自己掃搜尋引擎 |
| 更新 Notion 內容、寫入資料表、表格上色 | Phase 2，`POST /api/pipeline/notion/write` 回 501 |
| 垂直簡約時程圖、上傳、導入 Notion | Phase 2，`POST /api/pipeline/schedule/svg` 回 501 |
| 簡約期程 PPTX | Phase 2，`POST /api/pipeline/schedule/pptx` 回 501 |

頁面屬性（招商情形、編輯者、廠商數量、日期進度）本來就在 Notion 頁面上，不會放進 `updates`。

## 安裝與啟動

需要 Python 3.11 以上。在儲存庫根目錄執行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
python -m rag_server
```

服務預設聽 `http://127.0.0.1:8000`。狀態頁開 `http://127.0.0.1:8000/`（同樣內容在 `/ui`），分成「可用」與「尚未完成／Phase 2」。沒設 `RAG_CASES_ROOT` 會直接寫出來；「掃描內建示範樹」與「分析成 Notion JSON」用專案裡的合成案件，不需要 `D:`。API 文件在 `/docs`。健康檢查是 `GET /health`（`analysis_mode` 為 `extractive`，`local_llm_enabled` 表示有沒有接上電腦 B 的 Ollama）。

掃描頁 OCR（選用）：

```bash
pip install -e ".[ocr]"
# Debian/Ubuntu：sudo apt install tesseract-ocr tesseract-ocr-chi-tra tesseract-ocr-chi-sim
```

沒裝 Tesseract 時，幾乎沒有文字層的掃描頁會留下警告，不會假裝讀到內容。

## 把 CASES_ROOT 指到本機案件夾

在看得到該磁碟的 Windows 機器上設定，然後啟動這個服務。雲端代理讀不到 `D:`。

PowerShell：

```powershell
$env:RAG_CASES_ROOT = 'D:\HS公司資料\HS網站(新視野)\網站資料\最新消息\招商案件'
python -m rag_server
```

cmd 請加引號，因為路徑裡有括號：

```bat
set "RAG_CASES_ROOT=D:\HS公司資料\HS網站(新視野)\網站資料\最新消息\招商案件"
```

也可以在每次請求的 JSON 裡傳 `root`，不必寫進環境變數。

## 兩台電腦

- **電腦 A** 放案件 PDF（`RAG_CASES_ROOT` 指到 `D:\…\招商案件`），並在這台跑 RAG 服務。
- **電腦 B** 只跑 Ollama。A 用 Tailscale 連過去。不要把 Ollama 開到公網；綁位要讓 Tailscale 同伴連得到，不能只聽 `127.0.0.1`。
- 在 A 的 `.env` 設 `RAG_LLM_BASE_URL=http://<B 的 Tailscale IP>:11434/v1` 與 `RAG_LLM_MODEL=<模型名稱>`。客戶端會再接上 `/chat/completions`。
- 不要改成服務跑在 B、檔案留在 A。檔案共用以後再說。

## templateBlocks

Markdown 匯出只有 ❯ 標題順序，沒有 block id，表格列也會被收掉。不能拿那份檔案當 `templateBlocks`。

正式執行要先用 Notion API 取出該頁的 blocks，原樣放進 `templateBlocks`。測試用的合成骨架在 `tests/fixtures/template_blocks.json`：標題對得上 ❯計畫概述到 ❯申請人資格（含 ❯期程規範），id 都是假的。模板裡「一般資格」列已經是「單一申請人｜合作聯盟(3家以內)」時，不會再寫一次。

Ollama 只吃這支分析。在電腦 A 的 `.env` 設（IP 換成電腦 B 的 Tailscale 位址）：

```text
RAG_LLM_BASE_URL=http://<B 的 Tailscale IP>:11434/v1
RAG_LLM_MODEL=<模型名稱>
```

`RAG_LLM_API_KEY` 可留空。沒設 URL 或模型時，`/api/notion/analyze` 只用抽取式，不會連到 B。`/api/cases/{id}/query` 與 `/analyze` 永遠是抽取式。

## 範例請求

掃描、整棵寫入索引、合併單一案件文字：

```bash
curl -s -X POST http://127.0.0.1:8000/api/pipeline/detect \
  -H 'Content-Type: application/json' \
  -d '{"root":"D:\\HS公司資料\\HS網站(新視野)\\網站資料\\最新消息\\招商案件"}'

curl -s -X POST http://127.0.0.1:8000/api/pipeline/ingest \
  -H 'Content-Type: application/json' \
  -d '{"root":"D:\\HS公司資料\\HS網站(新視野)\\網站資料\\最新消息\\招商案件"}'

curl -s -X POST http://127.0.0.1:8000/api/pipeline/extract \
  -H 'Content-Type: application/json' \
  -d '{"root":"...\\招商案件","caseFolder":"...\\1.都市更新\\00基隆市\\基隆市七堵區一德新村公辦都更案"}'
```

合併文字沿用契約裡的分隔：`========== 文件 N ==========`、文件類型、檔案名稱。

分析（`templateBlocks` 換成該頁 API 取回的 blocks；下面只示意欄位）：

```bash
curl -s -X POST http://127.0.0.1:8000/api/notion/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "caseName": "示範市都更案",
    "pageId": "從 Notion 頁面複製的 page id",
    "pdfText": "計畫概述：……",
    "templateBlocks": [
      {"id": "真實的 block id", "type": "heading_2", "text": "❯計畫概述"},
      {"id": "真實的 paragraph id", "type": "paragraph", "text": ""}
    ]
  }'
```

也可以不傳 `pdfText`，改傳已索引的 `caseId`，或傳 `caseFolder` 讓服務自己合併該案 PDF。

合成案例抽出的形狀（節錄）：

```json
{
  "pageId": "page_demo_0001",
  "caseName": "示範市都更案",
  "baseTable": {
    "bases": [{
      "name": "示範更新基地",
      "location": "示範市示範區示範路與示範街所圍街廓",
      "scope": "示範段1、2地號等2筆土地",
      "area": "800平方公尺",
      "ownership": "• 私人（一般私人共10位）：100%"
    }]
  },
  "landUseTable": {
    "zones": [{
      "name": "第三種住宅區",
      "intensity": "建蔽率60%、容積率200%",
      "volumeItems": "☑ 都市更新容積獎勵\n☑ 綠建築容積獎勵\n☒ 容積移轉"
    }]
  }
}
```

`updates` 另含概述、周邊、退縮與倍數、同意比例、定位條列、兩列權利義務、期程（不含保證金）、財務條件，以及申請人資格各列。權利義務只填 PDF 真的有的列，不會補滿 6 列。

招標完成清單檔 `completed-cases.json`：

```json
{
  "cases": [{
    "case_name": "案名",
    "city": "縣市",
    "case_type": "公辦都更",
    "notion_page_id": "",
    "winning_applicant": "",
    "status": "招標完成",
    "source_urls": ["https://允許抓取的新聞或政府開放資料網址"]
  }]
}
```

CSV 欄位同名，`source_urls` 以空白分隔。設好 `RAG_COMPLETED_CASES_FILE` 後：

```bash
curl -s http://127.0.0.1:8000/api/completed
curl -s -X POST http://127.0.0.1:8000/api/completed/research \
  -H 'Content-Type: application/json' \
  -d '{"case_name":"案名","source_urls":["https://允許抓取的頁面"]}'
```

舊的單案問答仍在：`POST /api/cases`、上傳 PDF 到 `/api/cases/{id}/documents`、`/query`、`/analyze`、`/crawl`。爬蟲只抓公開 http(s)、讀 robots.txt、自報 User-Agent、請求之間等待，不登入、不偽裝瀏覽器、不解 CAPTCHA，也不抓本機或內網位址。

## 測試

測試用假嵌入，不下載模型，不連外網，也不使用真實案件內文。資料夾骨架在 `tests/fixtures/cases_root/`。

```bash
source .venv/bin/activate
pytest
```

## 索引怎麼存

單一程序、本機 SQLite（`RAG_DATA_DIR/index.sqlite`）。案件列會記下階段、類型、縣市與資料夾路徑；文件列會記下種類與來源路徑。同一路徑重跑 ingest 不會重複寫入。
