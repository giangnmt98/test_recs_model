#!/bin/bash

# === Define color codes ===
COLOR_YELLOW="\033[1;33m"  # Yellow for informational text
COLOR_RED="\033[1;31m"     # Red for errors or warnings
COLOR_GREEN="\033[1;32m"   # Green for successes
COLOR_RESET="\033[0m"      # Reset to default terminal color

# Separator for formatting
SEPARATOR="========================================"

# === Usage Function ===
if [ -z "$1" ]; then
    echo -e "${COLOR_YELLOW}Usage:${COLOR_RESET} ${0} <tag_name>"
    exit 1
fi

# === Variable containing the tag name ===
TAG_NAME="$1"

echo -e "${COLOR_YELLOW}${SEPARATOR}${COLOR_RESET}"
echo -e "${COLOR_GREEN}Processing tag:${COLOR_RESET} ${TAG_NAME}"
echo -e "${COLOR_YELLOW}${SEPARATOR}${COLOR_RESET}"

# === Check if the tag exists locally ===
if ! git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
    echo -e "${COLOR_RED}Tag '${TAG_NAME}' does not exist locally.${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}Skipping local deletion.${COLOR_RESET}"
else
    # Delete the local tag
    echo -e "${COLOR_YELLOW}Deleting local tag '${TAG_NAME}'...${COLOR_RESET}"
    git tag -d "$TAG_NAME"
    echo -e "${COLOR_GREEN}Local tag '${TAG_NAME}' deleted successfully.${COLOR_RESET}"
fi

# === Delete the tag on remote ===
echo -e "${COLOR_YELLOW}Deleting remote tag '${TAG_NAME}'...${COLOR_RESET}"
git push origin --delete "$TAG_NAME"
echo -e "${COLOR_GREEN}Remote tag '${TAG_NAME}' deleted successfully.${COLOR_RESET}"

# === Recreate the tag ===
echo -e "${COLOR_YELLOW}Creating new tag '${TAG_NAME}'...${COLOR_RESET}"
git tag -a "$TAG_NAME" -m "Recreated tag $TAG_NAME"
echo -e "${COLOR_GREEN}New tag '${TAG_NAME}' created successfully.${COLOR_RESET}"

# === Push the new tag to the remote repository ===
echo -e "${COLOR_YELLOW}Pushing new tag '${TAG_NAME}' to remote repository...${COLOR_RESET}"
git push origin "$TAG_NAME"

# === Final success message ===
echo -e "${COLOR_GREEN}Tag '${TAG_NAME}' has been successfully updated!${COLOR_RESET}"
