#!/bin/bash

# === CONFIGURATION ===
REPO_NAME="VCM_ST-NPP"
CLEANED_REPO="${REPO_NAME}-clean.git"
REMOTE_URL="https://github.com/NgocMinhUET/${REPO_NAME}.git"
BFG_JAR="bfg.jar"  # Ensure this file is in the same directory

# === STEP 1: Clone mirror repo ===
echo "Cloning mirror of the repository..."
git clone --mirror "$REMOTE_URL" "$CLEANED_REPO"

# === STEP 2: Remove large blobs using BFG ===
echo "Running BFG to delete blobs > 100MB..."
java -jar "$BFG_JAR" --strip-blobs-bigger-than 100M "$CLEANED_REPO"

# === STEP 3: Clean Git reflog and garbage collect ===
cd "$CLEANED_REPO"
echo "Running garbage collection..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# === STEP 4: Force push cleaned repo back to GitHub ===
echo "Pushing cleaned repository to GitHub..."
git push --force

echo "==== CLEANING COMPLETE ===="
echo "If collaborators are using this repo, they must re-clone from GitHub."
