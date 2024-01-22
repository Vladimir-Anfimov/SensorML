type Risk = [string, number];

type Data = {
  images: string[];
  risk: Risk[];
};

type PlotData = {
  prophet: Data;
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
