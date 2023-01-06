
export default function WalkOptions({
  noNoises,
  setNoNoises,
  slideStateChange
}) {
  return (
    <div className="extraOptions">
      <h3 className="fullRow optionHeader">Walk Specific Options</h3>
      <div className='alignCenter'>
          <p>No. Noises: <span id="demo">{noNoises}</span></p>
          <input type="range" min="1" step="1" max="30" value={noNoises} className='slider' id="myRange" onChange={e => slideStateChange(e, setNoNoises)} />
      </div>

    </div>
  )
}