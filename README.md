# FFXIV Prism

FFXIV Prism is a local Python project for matching an uploaded outfit image to a close Final Fantasy XIV glamour plan.

The first milestone is a local visual dataset of gear pieces, with Eorzea Collection as the primary visual source and XIVAPI as the structured game-data supplement.

## Current Scope

- No frontend.
- Runs locally from an IDE or Python entry point.
- Build the dataset before attempting image-to-glamour matching.

## Project Docs

- [Project phases](docs/project_phases.md)
- [Data collection plan](docs/data_collection_plan.md)

## Local Commands

Use the local Anaconda `dataprocess` environment:

```powershell
& 'C:\Softwares\Anaconda3\envs\dataprocess\python.exe' scripts\pilot_collect_eorzea.py --help
& 'C:\Softwares\Anaconda3\envs\dataprocess\python.exe' scripts\discover_eorzea_gearsets.py --help
& 'C:\Softwares\Anaconda3\envs\dataprocess\python.exe' scripts\build_eorzea_dataset_index.py
```

The collector respects robots.txt by default. If a source disallows automated fetches, stop and choose a permitted collection path before using `--ignore-robots`.

Canonical images come from `gearsets.eorzeacollection.com` static gearset files. Player-uploaded glamour pages are intentionally excluded from the dataset.

The collector keeps full-set `front`/`back` images and per-slot `head/body/hands/legs/feet` images. If a page exposes role variants in the same series, such as `fending`, `casting`, and `healing`, the pilot collects those role-specific gearsets too.
