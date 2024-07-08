import React from 'react';
import styled from 'styled-components';

const Container = styled.div`
  /*margin-bottom: 20px;*/
`;

const ResultItem = styled.div`
  /*background-color: #f8f9fa;
  border: 1px solid #cccccc;
  padding: 10px;
  border-radius: 5px;
  margin-bottom: 10px;*/
`;

const QueryResult = ({ response }) => {
    return (
      <Container>
        {response && response.map((result, index) => (
          <ResultItem key={index}>
            
          </ResultItem>
        ))}
      </Container>
    );
  };
  
export default QueryResult;
