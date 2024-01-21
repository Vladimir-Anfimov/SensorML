import { FileIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';

function Dropzone() {
  const [file, setFile] = useState<File | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFile(acceptedFiles[0]);
  }, []);

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
            <p>Drop the files here ...</p>
          ) : (
            <>
              <FileIcon className='mx-auto h-16 w-16 mb-4 ' color='#555' />
              <p>Drag 'n' drop a csv file here, or click to select a file</p>
            </>
          )}
        </div>
      </form>
      <p>{file?.name}</p>
    </>
  );
}

export { Dropzone };
