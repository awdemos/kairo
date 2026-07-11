# kairo

## Deployment

Observed deployment configuration:

- Container (`Dockerfile`) — build with `docker build -t <image> .`
- GitHub Actions workflows in `.github/workflows`

General redeploy process:

1. Commit and push changes to the default branch.
2. Trigger the relevant CI/CD pipeline or run the documented deploy command.
3. If the project is served via GitHub Pages, the site redeploys automatically after the push.
