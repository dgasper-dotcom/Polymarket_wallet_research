# Manual Seed Copy-Ready Watchlist

This watchlist uses the corrected standard: a wallet does not need to be a long-hold model. It only needs to appear copyable for a follower and not look HFT / execution-only.

## Counts
- `copy_ready`: `8`
- `monitor`: `2`
- `avoid`: `4`

## Copy Ready
- `0x53_candidate` `0x53ecc53e7a69aad0e6dda60264cc2e363092df91` | rationale `best_overall_copyable_candidate` | PMA copy30 `32953.89245105571` | local combined `22895.00056664452` | fast-exit30 `0.0`
- `sbinnala` `0xc483ee2ce773ae281131382ecc6285c968b88ac8` | rationale `copyable_and_not_hft` | PMA copy30 `16371.906666161754` | local combined `-646.1556949106637` | fast-exit30 `None`
- `SnowballHustle` `0xe36296a42555b95e95880412387e954d84b0bd00` | rationale `copyable_and_not_hft` | PMA copy30 `15644.365709365387` | local combined `8356.545983478507` | fast-exit30 `None`
- `MikeMoore` `0x5d2f49295387e01a49f0a3e59449ceed791c4adb` | rationale `copyable_and_not_hft` | PMA copy30 `15236.546211421617` | local combined `-3310.298811250883` | fast-exit30 `None`
- `PetrGrepl` `0xe7590338d435112c032e3ea51ff3d08a27a1e7ca` | rationale `copyable_and_not_hft` | PMA copy30 `5820.7732841247` | local combined `3701.9603872077632` | fast-exit30 `None`
- `ELICHOU` `0x312bcca3bc77bdc1d37dc6db5b9c1493de61cafe` | rationale `copyable_and_not_hft` | PMA copy30 `3381.7821593249446` | local combined `-2599.2862328085503` | fast-exit30 `None`
- `aikko` `0x68d1b156197fc516c56fc95d325b3716322c3c4d` | rationale `copyable_and_not_hft` | PMA copy30 `2656.9213393291398` | local combined `-1382.6302417296617` | fast-exit30 `None`
- `RobertoRubio` `0x3c4c03892f47d3166ee049a48a73d4743a17dd95` | rationale `copyable_and_not_hft` | PMA copy30 `1282.9768365382733` | local combined `-7176.437382014097` | fast-exit30 `None`

## Monitor
- `0x77_candidate` `0x77fd7aec1952ea7d042a6eec83bc4782f67db6c8` | rationale `copy_signal_conflict_needs_more_validation` | PMA copy30 `-4701.401435508163` | local combined `933.0762682187337` | fast-exit30 `0.0`
- `Kickstand7` `0xd1acd3925d895de9aec98ff95f3a30c5279d08d5` | rationale `crowded_reference_not_primary` | PMA copy30 `None` | local combined `73330.36697974277` | fast-exit30 `None`

## Avoid
- `Melody626` `0xecaa8806a9a05049d7d5260a33dc924220e377a9` | rationale `copyability_failed` | PMA copy30 `-128192.28254718459` | local combined `-36499.353175220014` | fast-exit30 `None`
- `Gambler1968` `0x7a6192ea6815d3177e978dd3f8c38be5f575af24` | rationale `insufficient_copyable_edge` | PMA copy30 `None` | local combined `28.353` | fast-exit30 `None`
- `ppp22232` `0x5d7639ed5fd3c5ebf0fc69d548e1046acd0f9168` | rationale `insufficient_copyable_edge` | PMA copy30 `None` | local combined `-1693.4431558988877` | fast-exit30 `None`
- `LuckyNFT444` `0x4bac379da2f29d87c01ff737843e396a2cec02b1` | rationale `insufficient_copyable_edge` | PMA copy30 `None` | local combined `-4556.461602265591` | fast-exit30 `0.0`

## Investor Readiness
- `investor_ready`: `True`
- The current copy-ready subset passed the full-history and OOS split checks used in this report.

## Exploratory Best Subset Search
- This is exploratory only and should not be treated as clean validation.
- Best subset by OOS count: `0xc483ee2ce773ae281131382ecc6285c968b88ac8,0xe36296a42555b95e95880412387e954d84b0bd00,0x5d2f49295387e01a49f0a3e59449ceed791c4adb,0xe7590338d435112c032e3ea51ff3d08a27a1e7ca,0x312bcca3bc77bdc1d37dc6db5b9c1493de61cafe`
- Positive OOS splits: `3`
- Full-history 30s PnL: `56455.374030398554`
- Positive-contribution concentration: `0.2899973111035683`
