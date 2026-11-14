# imagene-models

[Imagene](https://github.com/seiku77/imagene) アプリ用に、Stable Diffusion 1.5系モデルを MediaPipe Image Generator 形式に変換したものを置くリポジトリです。

- 変換は GitHub Actions(`.github/workflows/convert.yml`)で行い、結果を Releases に置きます
- 変換スクリプト: [google-ai-edge/mediapipe-samples](https://github.com/google-ai-edge/mediapipe-samples) の image_generation converter
- 2GBを超えるzipは `*.zip.part00`, `*.zip.part01` … に分割します(順に連結すると元のzipになります)
- `*.manifest.json` に URL・サイズ・SHA-256 を書いています

## ライセンス

各モデルは **CreativeML OpenRAIL-M** ライセンスです。利用制限(添付書類A)を含む全文を、各リリースに同梱しています。元モデルの作者に帰属します。
