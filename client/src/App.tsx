import './App.css';
import { Button } from './components/ui/button';
import { useNavigate } from 'react-router-dom';
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from './components/ui/carousel';

const heatmaps = [
  './public/images/heatmap-lumina.png',
  './public/images/heatmap-pres.png',
  './public/images/heatmap-stemp1.png',
  './public/images/heatmap-stemp2.png',
  './public/images/heatmap-temp1.png',
  './public/images/heatmap-temp2.png',
  './public/images/heatmap-umid.png',
];

const predictions = [
  './public/images/prophet1.png',
  './public/images/prophet2.png',
  './public/images/prophet3.png',
];

function App() {
  const navigate = useNavigate();

  return (
    <>
      <h1 className='text-5xl font-bold mb-8 mt-8'>How are your tomatoes?</h1>
      <p className='text-xl mb-16'>
        Check out our app to check the health of your tomatoes. <br />
        We trained our models on 2 years worth of data so you can see how
        healthy are your tomatoes.
      </p>
      <Button
        variant='default'
        className='mb-16 text-lg font-bold py-6 px-8'
        onClick={() => navigate('/diagnose')}
      >
        Try with your data
      </Button>
      <h2 className='text-4xl font-bold text-start mb-8'>
        Take a look at our results:
      </h2>
      <div className='flex flex-col flex-0 mb-24'>
        <h3 className='text-3xl font-bold text-start my-8'>Heatmaps:</h3>
        <Carousel className='mx-20'>
          <CarouselContent>
            {heatmaps.map((heatmap) => (
              <CarouselItem>
                <img
                  src={heatmap}
                  alt={heatmap.split('-')[1].split('.')[0]}
                  className='mb-8 inline-block'
                />
              </CarouselItem>
            ))}
          </CarouselContent>
          <CarouselPrevious />
          <CarouselNext />
        </Carousel>
        <h3 className='text-3xl font-bold text-start my-8'>Predictions:</h3>
        <Carousel className='mx-20'>
          <CarouselContent>
            {predictions.map((prediction) => (
              <CarouselItem>
                <img
                  src={prediction}
                  alt='prediction'
                  className='mb-8 inline-block'
                />
              </CarouselItem>
            ))}
          </CarouselContent>
          <CarouselPrevious />
          <CarouselNext />
        </Carousel>
        <h3 className='text-3xl font-bold text-start my-8'>
          Data correlation:
        </h3>
        <img src='./public/images/correlation.png' alt='data correlation' />
      </div>
    </>
  );
}

export default App;
