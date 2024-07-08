import React, { useState } from 'react';
import axios from 'axios';
import styled from 'styled-components';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  margin-bottom: 20px;
`;

const DropArea = styled.div`
  border: 2px dashed #cccccc;
  padding: 20px;
  text-align: center;
  cursor: pointer;
  width: 100px;
  margin-bottom: 10px;
`;

const UploadButton = styled.label`
  background-color: #4CAF50;
  color: white;
  padding: 10px 20px;
  border: none;
  cursor: pointer;
  border-radius: 5px;
`;

const FileUpload = ({ addMessage }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select a file.');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      //addMessage(`File uploaded successfully: ${file.name}`, false);
      alert('File uploaded successfully');
      setFile(null); // Clear selected file after successful upload
    } catch (error) {
      alert('Failed to upload file.');
      console.error(error);
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setFile(event.dataTransfer.files[0]);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  return (
    <Container>
      <DropArea
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        {file ? (
          <p>File selected: {file.name}</p>
        ) : (
          
            <UploadButton htmlFor="file">upload</UploadButton> 
           
        )}
        <input
          type="file"
          id="file"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
      </DropArea>
      {file && (
        <div>
          <UploadButton disabled={uploading} onClick={handleUpload}>
            {uploading ? 'Uploading...' : 'Upload'}
          </UploadButton>
        </div>
      )}
    </Container>
  );
};

export default FileUpload;
