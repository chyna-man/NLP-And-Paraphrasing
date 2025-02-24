import random

# Define intents
intents = {
    "Set a Timer": [
        "Set a timer for 5 minutes.",
        "Can you start a 5-minute countdown?",
        "I need a 5-minute timer.",
        "Create a timer for 5 minutes.",
        "Please begin a countdown of 5 minutes.",
        "Start a timer lasting 5 minutes.",
        "Set an alarm for five minutes from now.",
        "Can you count down 5 minutes for me?",
        "Begin a timer for 5 minutes.",
        "I need an alarm to go off in 5 minutes.",
        "Activate a five-minute timer.",
        "Countdown for 5 minutes, please.",
        "Start counting down from 5 minutes.",
        "Set my clock for a 5-minute alert.",
        "Create a five-minute countdown.",
        "Can you time 5 minutes for me?",
        "Set a reminder for 5 minutes later.",
        "I need a stopwatch for 5 minutes.",
        "Start an alarm in 5 minutes.",
        "Launch a 5-minute timer.",
        "Put a timer on for 5 minutes.",
        "Time 5 minutes for me.",
        "I need an alert in five minutes.",
        "Can you remind me in 5 minutes?",
        "Give me a countdown for five minutes.",
        "Please activate a 5-minute alarm.",
        "Time me for 5 minutes.",
        "I want a timer to ring in 5 minutes.",
        "Notify me when 5 minutes pass.",
        "Let me know when 5 minutes are up."
    ],
    "Play Music": [
        "Play some music.",
        "Can you put on a song?",
        "I want to hear some tunes.",
        "Start playing a song for me.",
        "Play my favorite music.",
        "Can you queue up some music?",
        "Turn on the music, please.",
        "I’d like to listen to some songs.",
        "Put on some background music.",
        "Begin playing my playlist.",
        "Start my daily mix.",
        "Play some relaxing songs.",
        "Find a good song and play it.",
        "Can you shuffle my playlist?",
        "I need some music to focus.",
        "Turn on my go-to tracks.",
        "Launch my favorite playlist.",
        "Play something upbeat.",
        "Shuffle my favorite songs.",
        "Play some instrumental music.",
        "Let’s hear some jazz.",
        "Start a random song.",
        "Can you DJ for me?",
        "Begin my road trip playlist.",
        "Play something trending.",
        "Find a song that suits my mood.",
        "Play the latest pop song.",
        "I need some calming sounds.",
        "Turn on some party music.",
        "Surprise me with a song."
    ],
    "Check the Weather": [
        "What's the weather like today?",
        "Give me a weather update.",
        "How's the weather outside?",
        "Can you check the forecast?",
        "Tell me if it's going to rain.",
        "Will it be sunny today?",
        "Check the weather for me.",
        "Is it cold outside?",
        "Do I need an umbrella today?",
        "What’s the temperature right now?",
        "Tell me today’s weather forecast.",
        "Give me the latest weather conditions.",
        "How’s the sky looking today?",
        "Will it be warm today?",
        "Is it windy outside?",
        "What’s the forecast for this evening?",
        "How’s the humidity today?",
        "Check the weather for the weekend.",
        "Is it snowing outside?",
        "Tell me if it’s going to storm.",
        "What’s the weather for tomorrow?",
        "Is there a heatwave expected?",
        "What’s the UV index today?",
        "How’s the air quality right now?",
        "Tell me if it’s foggy today.",
        "Is it safe to go outside?",
        "What’s the wind speed currently?",
        "Is it likely to rain this afternoon?",
        "Can you check the hourly forecast?",
        "Tell me if I should bring a jacket."
    ],
    "Control Smart Home Devices": [
        "Turn off the living room lights.",
        "Can you dim the bedroom lights?",
        "Turn on the kitchen lights.",
        "Adjust the thermostat to 72 degrees.",
        "Close the blinds in the bedroom.",
        "Increase the fan speed in the living room.",
        "Turn on the air conditioner.",
        "Lower the temperature by two degrees.",
        "Switch off all the lights.",
        "Lock the front door.",
        "Check if the windows are open.",
        "Turn off the TV.",
        "Turn on the heater.",
        "Change the thermostat to cool mode.",
        "Switch the lights to warm white.",
        "Set the brightness to 50%.",
        "Turn on the security cameras.",
        "Close the garage door.",
        "Activate the home alarm system.",
        "Turn off all appliances.",
        "Turn on the humidifier.",
        "Check if the oven is still on.",
        "Enable night mode for the lights.",
        "Adjust the living room curtains.",
        "Lower the blinds halfway.",
        "Turn off the sound system.",
        "Put the coffee maker on.",
        "Set the lights to reading mode.",
        "Disable all smart devices.",
        "Activate sleep mode on all electronics."
    ],
    "Send a Message": [
        "Send a message to John.",
        "Can you text Sarah?",
        "Write a message to my friend.",
        "Send a quick text to Dad.",
        "I need to message my boss.",
        "Send a WhatsApp message to Lisa.",
        "Email my colleague about the meeting.",
        "Shoot a text to Mike.",
        "Send a DM on Instagram.",
        "Reply to Jane’s last message.",
        "Forward this message to Tom.",
        "Send a voice message to Alex.",
        "Text Mom that I'm on my way.",
        "Write a new email to my professor.",
        "Send a Slack message to my team.",
        "Can you message my partner?",
        "Send a reminder to my assistant.",
        "Reply with 'Thanks' to Steve.",
        "Tell Jake I’ll call him later.",
        "Text my gym trainer about the schedule.",
        "Send a message to my group chat.",
        "Forward the latest update to HR.",
        "Notify my manager that I’m sick today.",
        "Message my brother on Facebook.",
        "Send an SMS to my landlord.",
        "Ask Lisa if she’s free for lunch.",
        "Write ‘See you soon’ to Mom.",
        "Text my best friend good luck.",
        "Message my partner ‘I love you.’",
        "Send a quick text: ‘Where are you?’"
    ]
}

# Generate dataset pairs
dataset = []
for intent, sentences in intents.items():
    for i in range(len(sentences) - 1):
        dataset.append((1, sentences[i], sentences[i + 1]))  # Positive paraphrase pair

# Generate negative pairs (different intents)
intent_list = list(intents.keys())
for _ in range(len(dataset) // 2):
    intent1, intent2 = random.sample(intent_list, 2)
    sent1 = random.choice(intents[intent1])
    sent2 = random.choice(intents[intent2])
    dataset.append((0, sent1, sent2))  # Negative pair

# Convert to DataFrame
import pandas as pd
df_synthetic = pd.DataFrame(dataset, columns=["Label", "#1 String", "#2 String"])

# Display the generated dataset
import ace_tools as tools
tools.display_dataframe_to_user(name="Synthetic VA Dataset", dataframe=df_synthetic)
