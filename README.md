# AI Image Compliance Detection

Early-stage prototype for AI-generated image detection and EU AI Act Article 50(4) disclosure support. **Not legal advice.**

## Architecture (planned)
Image → detection → VLM description → subclassification → legal RAG → report generation → citation verification.

## Quick start (Google Colab)
1. Create a GitHub repository and upload the contents of this starter project to its root.
2. Open `notebooks/00_setup.ipynb` in Google Colab.
3. Set `GITHUB_REPO` in the first code cell to `your-user/your-repo`.
4. Add `HF_TOKEN` and `KAGGLE_API_TOKEN` to Colab Secrets (use `KAGGLE_USERNAME` if your Kaggle access method needs it). Grant notebook access.
5. Run all cells. The setup installs the package and runs a placeholder test.

## Image Source
https://commons.wikimedia.org/w/index.php?search=city&title=Special%3AMediaSearch&type=image&haslicense=unrestricted

## Status
A0 starter scaffold only. No detection or compliance inference is implemented yet. See `CLAUDE.md`.

## Security
Never commit tokens, credentials, generated model weights, or private images.
