#!/bin/sh
# Builds index.html (standalone) and dist/artifact.html (content-only, for claude.ai publishing)
cd "$(dirname "$0")/.."
mkdir -p dist
{ echo '<!doctype html>'; echo '<html lang="ko">'; echo '<head>'; echo '<meta charset="utf-8">'; echo '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'; cat src/head.html; echo '</head>'; echo '<body>'; cat src/body.html src/script.html; echo '</body>'; echo '</html>'; } > index.html
cat src/head.html src/body.html src/script.html > dist/artifact.html
echo "built index.html ($(wc -c < index.html) bytes)"
