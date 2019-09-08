# If a command fails then the deploy stops
set -e

printf "Removing .git directory from 6.844 lab files...\n"

# Remove git directories (no longer tracked on course git repo)
rm -rf lab*/.git/

printf "Done.\n\n"
