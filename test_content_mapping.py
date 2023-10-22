def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def compare_text_files(file1, file2):
    text1 = read_file("C:/Users/nacho/Documents/Programacion/webots_2023/rescate_laberinto_2023/archivostxt/room4small.txt")
    text2 = read_file("C:/Users/nacho/Documents/Programacion/webots_2023/rescate_laberinto_2023/archivostxt//room_4_mapping.txt")

    # Calculate the percentage of coincidence
    common_characters = sum(1 for char1, char2 in zip(text1, text2) if char1 == char2)
    percentage_coincidence = (common_characters / max(len(text1), len(text2))) * 100

    # Calculate the difference
    differences = [(i+1, char1, char2) for i, (char1, char2) in enumerate(zip(text1, text2)) if char1 != char2]

    return percentage_coincidence, differences

# Provide the file paths for comparison
file1_path = 'file1.txt'
file2_path = 'file2.txt'

percentage, differences = compare_text_files(file1_path, file2_path)

print(f"Percentage of Coincidence: {percentage:.2f}%")
"""print(f"Differences:")
for position, char1, char2 in differences:
    print(f"Position {position}: '{char1}' in file1, '{char2}' in file2")
"""