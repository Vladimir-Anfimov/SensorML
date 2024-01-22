import { getData } from '../api';
import { useState } from 'react';
import { Dropzone } from './components/composed/dropzone';
import { Badge } from './components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './components/ui/table';
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from './components/ui/carousel';

type Risks = {
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

type Images = {
  prophet: string[];
  lstm: string[];
  seq2seq: string[];
};

function Diagnose() {
  const [risks, setRisks] = useState<Risks[]>([]);
  const [images, setImages] = useState<Images>({
    prophet: [],
    lstm: [],
    seq2seq: [],
  });
  const [isLoading, setIsLoading] = useState(false);

  function getBadgeColor(value: number) {
    if (value < 15) {
      return 'bg-blue-500 hover:bg-blue-700';
    }
    if (value < 30) {
      return 'bg-green-500 hover:bg-green-700';
    }
    if (value < 45) {
      return 'bg-yellow-500 hover:bg-yellow-600';
    }
    if (value < 60) {
      return 'bg-orange-500 hover:bg-orange-600';
    }
    return 'bg-red-600 hover:bg-red-800';
  }

  async function callApi(file: File) {
    setIsLoading(true);
    const data = await getData(file);

    setRisks(data.risks);
    setImages({
      prophet: data.prophet.images,
      lstm: data.lstm.images,
      seq2seq: data.seq2seq.images,
    });
    setIsLoading(false);
  }

  return (
    <>
      <h1 className='text-5xl font-bold mb-16'>Diagnose</h1>
      <Dropzone onFileUpload={callApi} />
      {isLoading && (
        <div className='absolute top-0 left-0 w-full h-full bg-gray-700 bg-opacity-50 flex justify-center items-center'>
          <div className='animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-gray-900' />
        </div>
      )}
      {isLoading === false && risks.length !== 0 && (
        <>
          <section className='my-16'>
            <h2 className='text-3xl font-bold mt-16 mb-8 text-left'>Results</h2>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className='font-bold w-[6rem] text-left'>
                    Disease
                  </TableHead>
                  <TableHead className='font-bold text-center'>
                    Prophet
                  </TableHead>
                  <TableHead className='font-bold text-center'>LSTM</TableHead>
                  <TableHead className='font-bold text-center'>
                    Seq2Seq
                  </TableHead>
                  <TableHead className='font-bold text-center'>
                    Overall
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {risks.map((result) => (
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
                      {condition.minAirTemperature}-
                      {condition.maxAirTemperature} °C
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
            <h3 className='text-2xl font-bold mt-16 mb-8 text-left'>
              This is what Prophet says:
            </h3>
            <Carousel className='mx-20'>
              <CarouselContent>
                {images.prophet.map((image) => (
                  <CarouselItem>
                    <img
                      src={image}
                      alt='prophet'
                      className='mb-8 inline-block'
                    />
                  </CarouselItem>
                ))}
              </CarouselContent>
              <CarouselPrevious />
              <CarouselNext />
            </Carousel>
            <h3 className='text-2xl font-bold mt-16 mb-8 text-left'>
              This is what LSTM says:
            </h3>
            <Carousel className='mx-20'>
              <CarouselContent>
                {images.lstm.map((image) => (
                  <CarouselItem>
                    <img
                      src={image}
                      alt='prophet'
                      className='mb-8 inline-block'
                    />
                  </CarouselItem>
                ))}
              </CarouselContent>
              <CarouselPrevious />
              <CarouselNext />
            </Carousel>
            <h3 className='text-2xl font-bold mt-16 mb-8 text-left'>
              This is what Seq2Seq says:
            </h3>
            <Carousel className='mx-20'>
              <CarouselContent>
                {images.seq2seq.map((image) => (
                  <CarouselItem>
                    <img
                      src={image}
                      alt='prophet'
                      className='mb-8 inline-block'
                    />
                  </CarouselItem>
                ))}
              </CarouselContent>
              <CarouselPrevious />
              <CarouselNext />
            </Carousel>
          </section>
        </>
      )}
    </>
  );
}

export default Diagnose;
