import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu';
import { Button } from '../ui/button';
import { ChevronsUpDown } from 'lucide-react';
import { useState } from 'react';

type PlotFigureProps = {
  className?: string;
  filepaths: Map<string, string>;
};

const models = ['Prophet', 'LSTM', 'Seq2Seq'];

function PlotFigure(props: PlotFigureProps) {
  const { className, filepaths } = props;
  const [model, setModel] = useState(models[0]);

  return (
    <div
      className={`relative inline-block pt-10 rounded border border-gray-700 shadow ${className}`}
    >
      <img src={filepaths.get(model)} alt={model} />
      <ChooseModelDropdown
        className='absolute top-2 right-2'
        model={model}
        setModel={setModel}
      />
    </div>
  );
}

type ChooseModelDropdownProps = {
  className?: string;
  model: string;
  setModel: (model: string) => void;
};

function ChooseModelDropdown(props: ChooseModelDropdownProps) {
  const { className, model, setModel } = props;

  return (
    <div className={`${className}`}>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant='outline'>
            {model}
            <ChevronsUpDown className='ml-2 h-4 w-4 shrink-0 opacity-50' />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className='w-56'>
          <DropdownMenuLabel>Choose your model</DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuRadioGroup value={model} onValueChange={setModel}>
            {models.map((model) => (
              <DropdownMenuRadioItem key={model} value={model}>
                {model}
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}

export { PlotFigure };
