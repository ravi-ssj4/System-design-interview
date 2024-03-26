# Code and example of a paginated api call to get a list of messages for a channel that the user clicked in a slack app using Javascript

For a real-world scenario where you might need to paginate API responses for a messaging app similar to Slack, using Node.js and Express along with a MongoDB database for storing messages could be a practical approach. This example assumes you have MongoDB set up with a collection named `messages` that stores messages for different channels. Each message document might look something like this:

```json
{
  "_id": "unique_message_id",
  "channelId": "channel_id_here",
  "text": "Message content here",
  "createdAt": "ISODate string"
}
```

### Setting up Your Project

1. **Initialize a new Node.js project** if you haven't already:

```bash
npm init -y
```

2. **Install necessary packages**:

```bash
npm install express mongoose
```

### Implementing the API

Create a file named `server.js` and add the following code to set up a basic Express server with a paginated endpoint for fetching messages:

```javascript
const express = require('express');
const mongoose = require('mongoose');

const app = express();
const PORT = process.env.PORT || 3000;

// MongoDB connection URI - update with your connection string
const mongoURI = 'mongodb://localhost:27017/yourDatabaseName';

// Connect to MongoDB
mongoose.connect(mongoURI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.error(err));

// Define a schema for the messages
const messageSchema = new mongoose.Schema({
  channelId: String,
  text: String,
  createdAt: { type: Date, default: Date.now }
});

// Create a model from the schema
const Message = mongoose.model('Message', messageSchema);

// Endpoint to fetch paginated messages for a channel
app.get('/api/messages', async (req, res) => {
  try {
    const { channelId, page = 1, limit = 10 } = req.query;

    // Ensure channelId is provided
    if (!channelId) {
      return res.status(400).send('ChannelId is required');
    }

    // Convert page and limit to numbers
    const pageNum = parseInt(page);
    const limitNum = parseInt(limit);

    // Query the database
    const messages = await Message.find({ channelId })
                                  .sort({ createdAt: -1 }) // Sort by newest first
                                  .skip((pageNum - 1) * limitNum)
                                  .limit(limitNum);

    res.json(messages);
  } catch (error) {
    console.error('Failed to fetch messages:', error);
    res.status(500).send('An error occurred while fetching messages');
  }
});

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
```

### Running Your Server

1. Make sure MongoDB is running and that you've replaced `mongoURI` with your actual MongoDB connection URI.
2. Start your server with `node server.js`.
3. To test the endpoint, make a GET request to `http://localhost:3000/api/messages?channelId=yourChannelId&page=1&limit=10`, replacing `yourChannelId` with the actual ID of the channel you want to fetch messages for.

This setup provides a basic example of implementing a paginated API for fetching messages from a specific channel using Express and MongoDB. It sorts messages by their creation date in descending order, so newer messages are fetched first. You can adjust the sorting, filtering, and pagination logic according to your specific requirements.