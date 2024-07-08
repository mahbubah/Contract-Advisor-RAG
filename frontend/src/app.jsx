import React from 'react';
import FileUpload from './FileUpload';
import QueryInput from './QueryInput';
import QueryResult from './QueryResult';

function App() {
  return (
    <div>
      <h1>Document Query System</h1>
      <FileUpload />
      <QueryInput />
      {/* <QueryResult response={response} /> */}
    </div>
  );
}

export default App;
