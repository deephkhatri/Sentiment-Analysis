from transformers import pipeline

# Initialize the sentiment analysis pipeline from Hugging Face
sent_pl = pipeline('sentiment-analysis')


def main():
    print("Welcome to the SA - pipeline!")
    while True:
        # Take input from the user
        user_input = input("Enter your text: ")

        # Exit condition
        if user_input.lower() == 'exit':
            print("Exiting the pipeline. Goodbye!")
            break

        # Process input through the sentiment analysis pipeline
        result = sent_pl(user_input)
        print(f"Pipeline Output: {result}")

        # Ask if the user wants to continue
        continue_input = input("Do you want to continue? (yes/no): ").lower()
        if continue_input not in ['yes', 'y']:
            print("Thank you for using the sentiment analysis pipeline. Goodbye!")
            break


if __name__ == "__main__":
    main()