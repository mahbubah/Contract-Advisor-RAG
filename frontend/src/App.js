import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import QueryInput from './QueryInput';
import QueryResult from './QueryResult';
import FileUpload from './FileUpload';

const Container = styled.div`
  background-color: #f0f0f0;
  color: #333;
  font-family: Arial, sans-serif;
  padding: 30px;
  display: grid;
  grid-template-columns: 1fr 4fr;
  grid-template-rows: auto;
  gap: 10px;
`;

const ChatContainer = styled.div`
  grid-column: 2 / 4;
  grid-row: 1 / 2;
  padding: 10px;
  border-radius: 10px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
`;

const UploadContainer = styled.div`
  grid-column: 1 / 2;
  grid-row: 1 / 2;
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  width: 200px;
`;

const Message = styled.div`
  background-color: ${(props) => (props.fromUser ? '#007bff' : '#f8f9fa')};
  color: ${(props) => (props.fromUser ? '#fff' : '#333')};
  padding: 10px;
  border-radius: ${(props) => (props.fromUser ? '10px 10px 0 10px' : '10px 10px 10px 0')};
  margin-bottom: 10px;
  max-width: 80%;
  align-self: ${(props) => (props.fromUser ? 'flex-end' : 'flex-start')};
`;

const App = () => {
  const [messages, setMessages] = useState([]);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const handleQuerySubmit = async (query) => {
    try {
      const response = await axios.post('http://localhost:5000/query', { query });
      addMessage(query, true);
      addMessage(response.data.response, false);
    } catch (error) {
      console.error('Error fetching data:', error);
      // Handle error state if needed
    }
  };

  const addMessage = (content, fromUser) => {
    setMessages((prevMessages) => [
      ...prevMessages,
      { content, fromUser }
    ]);
  };

  return (
    <Container>
      <UploadContainer>
        <h2>Upload your file here</h2>
        <FileUpload addMessage={addMessage} />
      </UploadContainer>
      <ChatContainer>
        <h2>Contract Q&A</h2>
        <div style={{ maxHeight: '1000px', overflowY: 'auto', marginBottom: '20px' }}>
          {messages.map((message, index) => (
            <Message key={index} fromUser={message.fromUser}>
              {message.content}
            </Message>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <QueryInput handleQuerySubmit={handleQuerySubmit} />
        <QueryResult response={messages.filter(msg => !msg.fromUser).map(msg => msg.content)} />
      </ChatContainer>
    </Container>
  );
};

export default App;
