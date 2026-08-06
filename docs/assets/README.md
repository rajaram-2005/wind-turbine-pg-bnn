# AeroVigil social & release graphics

Brand assets for announcements, posts, and the repository social preview. All
share the deep-navy / teal palette and the "advisory only" positioning.

| File | Intended use | Format |
| --- | --- | --- |
| `social-card.png` | **GitHub social preview** (repo Settings → Social preview) and Open-Graph/Twitter link cards | 2:1 landscape |
| `release-banner.png` | **GitHub Release notes header** and blog/announcement hero for v1.0.0 | ~21:9 ultra-wide |
| `social-square.png` | **LinkedIn / X post image** highlighting the 45-day early-warning capability | 1:1 square |

## Usage notes

- Upload `social-card.png` under
  [repo Settings → Social preview](https://github.com/rajaram-2005/wind-turbine-pg-bnn/settings)
  so shared links render the card.
- Embed `release-banner.png` at the top of release notes:

  ```markdown
  ![AeroVigil v1.0.0](docs/assets/release-banner.png)
  ```

- Keep the "advisory only" caption visible when cropping `social-square.png` —
  it is part of the safety positioning, not decoration.
