import React, { useState } from 'react';
import styled from 'styled-components';

const InputContainer = styled.div`
  display: flex;
  margin-top: 10px;
`;

const Input = styled.input`
  flex: 1;
  padding: 10px;
  font-size: 16px;
  border: 1px solid #ccc;
  border-radius: 5px 0 0 5px;
`;

const SubmitButton = styled.button`
  background-color: #007bff;
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 0 5px 5px 0;
  cursor: pointer;
`;

const QueryInput = ({ handleQuerySubmit }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    if (query.trim() === '') return; // Prevent empty queries
    handleQuerySubmit(query);
    setQuery('');
  };

  return (
    <form onSubmit={handleSubmit}>
      <InputContainer>
        <Input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Type your query here..."
        />
        <SubmitButton type="submit">Send</SubmitButton>
      </InputContainer>
    </form>
  );
};

export default QueryInput;
