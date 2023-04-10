import React, { useState } from 'react';
import { Button, TextField, Typography, Box, Grid } from '@mui/material';

function App() {
  const [numTopics, setNumTopics] = useState(5);
  const [numWords, setNumWords] = useState(10);
  const [topics, setTopics] = useState([]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const response = await fetch(`https://us-central1-book-of-mormon-nlp.cloudfunctions.net/process_text`);
    const data = await response.json();
    setTopics(data.topics);
  };

  return (
    <Box m={4}>
      <Typography variant="h4" align="center" gutterBottom>
        Book of Mormon LDA
      </Typography>
      <form onSubmit={handleSubmit}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Number of Topics"
              variant="outlined"
              type="number"
              value={numTopics}
              onChange={(event) => setNumTopics(event.target.value)}
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Number of Words"
              variant="outlined"
              type="number"
              value={numWords}
              onChange={(event) => setNumWords(event.target.value)}
            />
          </Grid>
          <Grid item xs={12}>
            <Button type="submit" variant="contained" color="primary">
              Submit
            </Button>
          </Grid>
        </Grid>
      </form>
      <Box mt={4}>
        {topics.length > 0 && (
          <ul>
            {topics.map((topic, index) => (
              <li key={index}>
                <Typography variant="h6" gutterBottom>
                  Topic {index + 1}
                </Typography>
                <ul>
                  {topic.map((word, index) => (
                    <li key={index}>{word}</li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        )}
      </Box>
    </Box>
  );
}

export default App;
