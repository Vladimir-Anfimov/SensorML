import './App.css';
import { useState } from 'react';
import { PlotFigure } from './components/composed/plot-figure';
import { Button } from './components/ui/button';
import { useNavigate } from 'react-router-dom';

function App() {
  const navigate = useNavigate();

  const models = ['Prophet', 'LSTM', 'Seq2Seq'];
  const [model, setModel] = useState(models[0]);

  return (
    <>
      <h1 className='text-5xl font-bold mb-8 mt-16'>How are your tomatoes?</h1>
      <p className='text-xl mb-16'>
        Check out our app to check the health of your tomatoes. <br />
        We trained our models on 2 years worth of data so you can see how
        healthy are your tomatoes.
      </p>
      <Button
        variant='default'
        className='mb-32 text-lg font-bold py-6 px-8'
        onClick={() => navigate('/diagnose')}
      >
        Try with your data
      </Button>
      <div className='flex flex-col items-center mb-8'>
        <PlotFigure
          className='mb-8'
          model={model}
          setModel={setModel}
          models={models}
        />
        <PlotFigure
          className='mb-8'
          model={model}
          setModel={setModel}
          models={models}
        />
        <PlotFigure
          className='mb-8'
          model={model}
          setModel={setModel}
          models={models}
        />
        <PlotFigure
          className='mb-8'
          model={model}
          setModel={setModel}
          models={models}
        />
      </div>
    </>
  );
}

export default App;
