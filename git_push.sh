# Check if a commit message is provided
if [ -z "$1" ]; then
  echo "Usage: ./git_push.sh 'Your commit message'"
  exit 1
fi

# Add all changes
git add .

# Commit with the provided message
git commit -m "$1"

# Push to the remote repository
git push origin # Replace 'main' with your branch name if different

# End of script

