import { useState } from 'react';
import { Dropzone } from './components/composed/dropzone';
import { PlotFigure } from './components/composed/plot-figure';
import { Badge } from './components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './components/ui/table';
import { Button } from './components/ui/button';

type Result = {
  disease: string;
  prophet: number;
  lstm: number;
  seq2seq: number;
  overall: number;
};

type OptimumConditions = {
  disease: string;
  minAirTemperature: number;
  maxAirTemperature: number;
  minAirHumidity: number;
  maxAirHumidity: number;
};

const results: Result[] = [
  {
    disease: 'LateBlight',
    prophet: 90,
    lstm: 80,
    seq2seq: 70,
    overall: 80,
  },
  {
    disease: 'EarlyBlight',
    prophet: 28,
    lstm: 19,
    seq2seq: 32,
    overall: 26,
  },
  {
    disease: 'PowderyMildew',
    prophet: 63,
    lstm: 50,
    seq2seq: 78,
    overall: 64,
  },
  {
    disease: 'GrayMold',
    prophet: 72,
    lstm: 65,
    seq2seq: 80,
    overall: 72,
  },
  {
    disease: 'LeafMold',
    prophet: 21,
    lstm: 10,
    seq2seq: 25,
    overall: 20,
  },
];

const optimumConditions: OptimumConditions[] = [
  {
    disease: 'LateBlight',
    minAirTemperature: 10,
    maxAirTemperature: 24,
    minAirHumidity: 90,
    maxAirHumidity: 100,
  },
  {
    disease: 'EarlyBlight',
    minAirTemperature: 24,
    maxAirTemperature: 29,
    minAirHumidity: 90,
    maxAirHumidity: 100,
  },
  {
    disease: 'PowderyMildew',
    minAirTemperature: 22,
    maxAirTemperature: 30,
    minAirHumidity: 50,
    maxAirHumidity: 75,
  },
  {
    disease: 'GrayMold',
    minAirTemperature: 17,
    maxAirTemperature: 23,
    minAirHumidity: 90,
    maxAirHumidity: 100,
  },
  {
    disease: 'LeafMold',
    minAirTemperature: 21,
    maxAirTemperature: 24,
    minAirHumidity: 85,
    maxAirHumidity: 100,
  },
];

function Diagnose() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  function getBadgeColor(value: number) {
    if (value < 15) {
      return 'bg-blue-500 hover:bg-blue-700';
    }
    if (value < 30) {
      return 'bg-green-500 hover:bg-green-700';
    }
    if (value < 50) {
      return 'bg-yellow-500 hover:bg-yellow-600';
    }
    if (value < 75) {
      return 'bg-orange-500 hover:bg-orange-600';
    }
    return 'bg-red-600 hover:bg-red-800';
  }

  // HAICI
  async function callApi() {
    const api_endpoint = 'http://localhost:8000/diagnose';

    if (!selectedFile) {
      alert('BA INCARCA IAIC');
    }

    const formData = new FormData();
    formData.append('file', selectedFile as Blob);

    fetch(api_endpoint, {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
      })
      .catch((error) => {
        console.error(error);
      });
  }

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    if (event.target.files) {
      setSelectedFile(event.target.files[0]);
    }
  }

  return (
    <>
      <h1 className='text-5xl font-bold mb-16'>Diagnose</h1>
      <div>
        <Button
          variant='default'
          className='mb-32 text-lg font-bold py-6 px-8'
          onClick={callApi}
        >
          Try with your data
        </Button>
        <br />
        <input type='file' onChange={handleFileChange} accept='.csv' />
      </div>
      <Dropzone />
      <section className='my-16'>
        <h2 className='text-3xl font-bold mt-16 mb-8 text-left'>Results</h2>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className='font-bold w-[6rem] text-left'>
                Disease
              </TableHead>
              <TableHead className='font-bold text-center'>Prophet</TableHead>
              <TableHead className='font-bold text-center'>LSTM</TableHead>
              <TableHead className='font-bold text-center'>Seq2Seq</TableHead>
              <TableHead className='font-bold text-center'>Overall</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {results.map((result) => (
              <TableRow key={result.disease}>
                <TableCell className='w-[6rem] text-left'>
                  {result.disease}
                </TableCell>
                <TableCell className='text-center'>
                  <Badge className={getBadgeColor(result.prophet)}>
                    {result.prophet}%
                  </Badge>
                </TableCell>
                <TableCell className='text-center'>
                  <Badge className={getBadgeColor(result.lstm)}>
                    {result.lstm}%
                  </Badge>
                </TableCell>
                <TableCell className='text-center'>
                  <Badge className={getBadgeColor(result.seq2seq)}>
                    {result.seq2seq}%
                  </Badge>
                </TableCell>
                <TableCell className='text-center'>
                  <Badge className={getBadgeColor(result.overall)}>
                    {result.overall}%
                  </Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </section>
      <section className='my-16'>
        <h2 className='text-3xl font-bold mt-16 mb-8 text-left'>
          Optimum Conditions for Disease Manifastation
        </h2>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead />
              {optimumConditions.map((condition) => (
                <TableHead
                  className='font-bold text-center'
                  key={condition.disease}
                >
                  {condition.disease}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            <TableRow>
              <TableCell className='w-[10rem] text-left'>
                Air Temperature
              </TableCell>
              {optimumConditions.map((condition) => (
                <TableCell className='text-center' key={condition.disease}>
                  {condition.minAirTemperature}-{condition.maxAirTemperature} °C
                </TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell className='w-[10rem] text-left'>
                Air Humidity
              </TableCell>
              {optimumConditions.map((condition) => (
                <TableCell className='text-center' key={condition.disease}>
                  {condition.minAirHumidity}-{condition.maxAirHumidity} %
                </TableCell>
              ))}
            </TableRow>
          </TableBody>
        </Table>
      </section>
      <section className='my-16'>
        <h2 className='text-3xl font-bold mt-16 mb-8 text-left'>
          Here is how we think your tomatoes will do
        </h2>
        <PlotFigure
          className='mb-8'
          filepaths={
            new Map([
              ['Prophet', './public/images/prophet1.png'],
              ['LSTM', './public/images/prophet2.png'],
              ['Seq2Seq', './public/images/prophet3.png'],
            ])
          }
        />
      </section>
    </>
  );
}

export default Diagnose;
