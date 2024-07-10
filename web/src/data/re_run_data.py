import json

# Load the content of all_scores.json
with open("/Users/phanmanhtuan/work/anh_trinh/raid/web/src/data/all-scores.json", "r") as file:
    all_scores = json.load(file)

# Load the content of new_score.json
# with open("/Users/phanmanhtuan/work/anh_trinh/raid/leaderboard/submissions/detector1/results.json", "r") as file:
#     new_score = json.load(file)
#

# file_to_read = "/Users/phanmanhtuan/work/anh_trinh/raid/leaderboard/submissions/detector21/results.json"
file_to_read = "/Users/phanmanhtuan/work/anh_trinh/raid/leaderboard/submissions/detector30/results.json"
with open(file_to_read, "r") as file:
    new_score = json.load(file)

# Append the new score to the list of all scores
all_scores.append(new_score)

# Write the updated list back to all_scores.json
with open("/Users/phanmanhtuan/work/anh_trinh/raid/web/src/data/all-scores.json", "w") as file:
    json.dump(all_scores, file, indent=4)

print("The new score has been appended successfully.")
