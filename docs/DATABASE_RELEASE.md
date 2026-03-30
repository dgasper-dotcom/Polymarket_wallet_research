# Database Release

The main research database is too large to store directly in the Git repository.

GitHub repository storage is used for:
- source code
- tests
- lightweight result bundles

The full SQLite database snapshot is distributed separately as GitHub Release assets, split into multiple parts small enough for GitHub upload limits.

To reconstruct the database after downloading all parts:

```bash
cat polymarket_wallets.db.part-* > polymarket_wallets.db
```

Then verify the checksum using the included `SHA256SUMS.txt`.
