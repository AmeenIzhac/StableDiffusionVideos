

export default function DropdownOptions({
  isImg2Img,
  setisImg2Img,
}) {
  const dark = '#181818'
  const light = '#9A4EAE'
  const defaultStyle = {
      backgroundColor: 'transparent',
      color: light,
      borderTop: '1px solid ' + light
  }
  const chosenStyle = {
      backgroundColor: 'transparent',
      color: light,
      borderBottom: '3px solid',
      borderImageSource: ' linear-gradient(to right, #de93fa, #98a7f9)',
      borderImageSlice: '1'
  }

  const img2imgStyle = isImg2Img ? chosenStyle : defaultStyle 
  const walkStyle = !isImg2Img ? chosenStyle : defaultStyle 

  const chooseImg2Img = () => {
        setisImg2Img(true)
    }
    const chooseWalk = () => {
        setisImg2Img(false)
    }
    return (
      <div className="dropdownVidOption fullRow">
          <ul className="vidOptionContainer">
              <li className="img2img">
                  <button className="vidOptionBtn" style={img2imgStyle} onClick={chooseImg2Img}>img2img</button>
              </li>
              <li className="walk">
                  <button className="vidOptionBtn" style={walkStyle} onClick={chooseWalk}>Walk</button>
              </li>
          </ul>
      </div>
    )

}