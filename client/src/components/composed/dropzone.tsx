import { FileIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';

type DropzoneProps = {
  onFileUpload: (file: File) => void;
};

function Dropzone(props: DropzoneProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      props.onFileUpload(acceptedFiles[0]);
    },
    [props],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false,
  });

  return (
    <>
      <form>
        <div
          {...getRootProps({
            // make the border dashed
            className: 'border-2 border-gray-700 border-dashed rounded p-8',
          })}
        >
          <input {...getInputProps()} />
          {isDragActive ? (
            <>
              <FileIcon className='mx-auto h-16 w-16 mb-4 ' color='#555' />
              <p>Drop the files here ...</p>
            </>
          ) : (
            <>
              <FileIcon className='mx-auto h-16 w-16 mb-4 ' color='#555' />
              <p>Drag 'n' drop a csv file here, or click to select a file</p>
            </>
          )}
        </div>
      </form>
    </>
  );
}

export { Dropzone };
