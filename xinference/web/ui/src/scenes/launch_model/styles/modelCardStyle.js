const CARD_HEIGHT = 300
const CARD_WIDTH = 300

const styles = {
  container: {
    display: 'block',
    position: 'relative',
    width: `${CARD_WIDTH}px`,
    height: `${CARD_HEIGHT}px`,
    border: '1px solid #ddd',
    borderRadius: '20px',
    background: 'white',
    overflow: 'hidden',
  },
  containerSelected: {
    display: 'block',
    position: 'relative',
    width: `${CARD_WIDTH}px`,
    height: `${CARD_HEIGHT}px`,
    border: '1px solid #ddd',
    borderRadius: '20px',
    background: 'white',
    overflow: 'hidden',
    boxShadow: '0 0 2px #00000099',
  },
  descriptionCard: {
    position: 'relative',
    top: '-1px',
    left: '-1px',
    width: `${CARD_WIDTH}px`,
    height: `${CARD_HEIGHT}px`,
    border: '1px solid #ddd',
    padding: '20px',
    borderRadius: '20px',
    background: 'white',
  },
  parameterCard: {
    position: 'relative',
    top: `-${CARD_HEIGHT + 1}px`,
    left: '-1px',
    width: `${CARD_WIDTH}px`,
    height: `${CARD_HEIGHT}px`,
    border: '1px solid #ddd',
    padding: '20px',
    borderRadius: '20px',
    background: 'white',
  },
  drawerCard: {
    position: 'relative',
    padding: '20px 80px 0',
    minHeight: '100%',
    width: '60vw'
  },
  img: {
    display: 'block',
    margin: '0 auto',
    width: '180px',
    height: '180px',
    objectFit: 'cover',
    borderRadius: '10px',
  },
  h2: {
    margin: '10px 10px',
    fontSize: '20px',
  },
  p: {
    fontSize: '14px',
    padding: '0px 10px 15px 10px',
  },
  formContainer: {
    height: '80%',
    overflow: 'scroll',
    padding: '0 10px'
  },
  buttonsContainer: {
    position: 'absolute',
    bottom: '50px',
    left: '100px',
    right: '100px',
    display: 'flex',
    margin: '0 auto',
    marginTop: '15px',
    border: 'none',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  buttonContainer: {
    width: '45%',
    borderWidth: '0px',
    backgroundColor: 'transparent',
    paddingLeft: '0px',
    paddingRight: '0px',
  },
  buttonItem: {
    width: '100%',
    margin: '0 auto',
    padding: '5px',
    display: 'flex',
    justifyContent: 'center',
    borderRadius: '4px',
    border: '1px solid #e5e7eb',
    borderWidth: '1px',
    borderColor: '#e5e7eb',
  },
  instructionText: {
    fontSize: '12px',
    color: '#666666',
    fontStyle: 'italic',
    margin: '30px 0',
    textAlign: 'center',
  },
  slideIn: {
    transform: 'translateX(0%)',
    transition: 'transform 0.2s ease-in-out',
  },
  slideOut: {
    transform: 'translateX(100%)',
    transition: 'transform 0.2s ease-in-out',
  },
  iconRow: {
    position: 'absolute',
    bottom: '20px',
    left: '20px',
    right: '20px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  iconItem: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    margin: '20px',
  },
  boldIconText: {
    fontWeight: 'bold',
    fontSize: '1.2em',
  },
  muiIcon: {
    fontSize: '1.5em',
  },
  smallText: {
    fontSize: '0.8em',
  },
  tagRow: {
    margin: '2px 5px',
  },
}

export default styles