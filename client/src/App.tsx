import './App.css';

function App() {
  return (
    <>
      <h1>Polyglot 🦜 💬 🇬🇧 ➡️ 🏴‍☠️</h1>
      <p>Translate English into Pirate speak...</p>
      <input
        type='text'
        name='english'
        id='english'
        placeholder='Type English here...'
      />
      <button py-click='translate_english'>Translate</button>
      <div id='output'></div>
    </>
  );
}

export default App;
