import nltk

# Write this in terminal to install the necessary certificates
# /Applications/Python\ 3.12/Install\ Certificates.command

nltk.download('stopwords')
from nltk.corpus import stopwords

# Get English stopwords
stop_words = set(stopwords.words('turkish'))

# Print stopwords
print(stop_words)