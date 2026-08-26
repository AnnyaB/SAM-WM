# Mac migration + local UI preview

The ChatGPT response supplies the exact copy/push commands. The safety rules are:

1. back up `.env` outside the repo;
2. back up the existing `static/` UI outside the repo;
3. replace the research tree on `sam-wm-v1-mechanism-redesign`, not `main`;
4. restore only `.env` locally (never commit it);
5. do not restore/push UI until visual approval;
6. run `pytest` before any push;
7. review `git status` and verify `.env` is absent from tracked changes.
