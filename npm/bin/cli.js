#!/usr/bin/env node

const { execSync, spawnSync } = require("child_process");

function findPython() {
  for (const cmd of ["python3", "python"]) {
    try {
      const version = execSync(`${cmd} --version 2>&1`, {
        encoding: "utf-8",
      }).trim();
      const match = version.match(/Python (\d+)\.(\d+)/);
      if (match && parseInt(match[1]) >= 3 && parseInt(match[2]) >= 11) {
        return cmd;
      }
    } catch {}
  }
  return null;
}

function main() {
  const args = process.argv.slice(2);

  const python = findPython();
  if (!python) {
    console.error("Error: Python 3.11+ is required but not found.");
    console.error("Install Python from https://python.org");
    process.exit(1);
  }

  try {
    execSync(`${python} -c "import collivind"`, { stdio: "ignore" });
  } catch {
    console.log("Installing collivind-memory...");
    const installResult = spawnSync(
      python,
      ["-m", "pip", "install", "collivind-memory"],
      { stdio: "inherit" }
    );
    if (installResult.status !== 0) {
      console.error("Failed to install collivind-memory");
      process.exit(1);
    }
  }

  const result = spawnSync(python, ["-m", "collivind", ...args], {
    stdio: "inherit",
  });
  process.exit(result.status || 0);
}

main();
