type Risk = {
  disease: string;
  prophet: number;
  lstm: number;
  seq2seq: number;
  overall: number;
};

type Data = {
  images: string[];
};

type PlotData = {
  prophet: Data;
  lstm: Data;
  seq2seq: Data;
  risks: Risk[];
};

async function getData(file: File): Promise<PlotData> {
  const formData = new FormData();
  formData.append('file', file as Blob);

  const result = await fetch('http://localhost:8000/diagnose', {
    method: 'POST',
    body: formData,
  });
  const data: PlotData = await result.json();
  return data;
}

export { getData };
export type { PlotData };
