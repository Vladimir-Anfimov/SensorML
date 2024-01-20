import './App.css';
import { useState } from 'react';
import { Button } from './components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './components/ui/dropdown-menu';

function App() {
  const [model, setModel] = useState('Prophet');

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant='outline'>{model}</Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className='w-56'>
          <DropdownMenuLabel>Panel Position</DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuRadioGroup value={model} onValueChange={setModel}>
            <DropdownMenuRadioItem value='LSTM'>LSTM</DropdownMenuRadioItem>
            <DropdownMenuRadioItem value='Prophet'>
              Prophet
            </DropdownMenuRadioItem>
            <DropdownMenuRadioItem value='Seq2Seq'>
              Seq2Seq
            </DropdownMenuRadioItem>
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  );
}

export default App;
