# Put AstroAI on your iPhone

The simplest final demo is Streamlit Community Cloud. The repository already has a `requirements.txt` and `app.py` at the project root.

## GitHub

Create a repository, then upload all files from this project.

## Streamlit Community Cloud

Create a new app, select your GitHub repository, choose `app.py` as the main file, and deploy.

The resulting HTTPS page can be opened in Safari on an iPhone. The `st.camera_input` widget provides browser camera capture, while the upload widget provides a fallback for Photos/files.

## Local laptop test

Run:

```text
conda activate astroai
streamlit run app.py --server.address 0.0.0.0
```

Then open the laptop's LAN address from the phone. Photo upload may work locally, but direct camera capture can be rejected by mobile browsers on a non-HTTPS LAN page. For reliable camera testing, use the HTTPS deployment.
