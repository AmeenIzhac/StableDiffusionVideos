

export default function FrameSelect(props) {

    const srcs = props.srcs;
    const selectFunction = props.selectFunction;
    const getNewFrame = props.getNewFrame;

    // return a grid of four images with hover effects
    // TODO: make this a grid of 4 images with srcs as the src
    return (
        <div className="frameSelect" key={props.srcs}>
            {/* {srcs.map((src, index) => (
                <div className="frameContainer" key={index} onClick={() => selectFunction(index)}>
                    <img
                        src={'../assets/a.jpg'}
                        alt={`Frame ${index}`}
                        className="frameImage"
                    />
                    <div className="frameOverlay">
                        <div className="text">Select Frame</div>
                    </div>
                </div>
                ))} */}
            <p> helllo </p>
            {/* TODO: remove button after 4 frames show */}
            <div className="FrameContainer" onClick={() => getNewFrame()}>
                <button className="newFrameButton" title="Get new Frame"> + </button>
            </div>
        </div>
    );

}