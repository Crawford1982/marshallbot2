// static/js/chatbot.js
function sendMessage() {
  let userInput = document.getElementById("user_input").value.trim();
  if (userInput === "") {
    return;
  }

  let messagesDiv = document.getElementById("messages");

  // Display user's message
  let userMessage = document.createElement("div");
  userMessage.className = "user-message";
  userMessage.textContent = "You: " + userInput;
  messagesDiv.appendChild(userMessage);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;

  // Clear input field
  document.getElementById("user_input").value = "";

  // Send message to server
  fetch("/get_response", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message: userInput }),
  })
    .then((response) => response.json())
    .then((data) => {
      // Display bot's reply
      let botMessage = document.createElement("div");
      botMessage.className = "bot-message";
      botMessage.textContent = "Bot: " + data.reply;
      messagesDiv.appendChild(botMessage);
      messagesDiv.scrollTop = messagesDiv.scrollHeight;
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

// Allow sending message with Enter key
document
  .getElementById("user_input")
  .addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      sendMessage();
    }
  });
