# Menu embeddings exercise

A small exercise to learn text embeddings: each menu item is embedded once,
a question is embedded the same way, and the closest item (by cosine similarity)
is returned. Items below a similarity threshold (0.50) count as "no match".

Run: create a venv, `pip install -r requirements.txt`, put `OPENAI_API_KEY` in `.env`,
then `python main.py`.

Notes: 20 simple test questions, all passed. The threshold was picked on the same
questions, so the score is optimistic. An earlier version on a soccer FAQ scored much
lower, because harder questions need better matching.