import { RocketLaunchOutlined, UndoOutlined } from '@mui/icons-material'
import { Box, Chip, CircularProgress } from '@mui/material'
import React, { useContext, useEffect, useState } from 'react'
import { v1 as uuidv1 } from 'uuid'

import { ApiContext } from '../../components/apiContext'

const CARD_HEIGHT = 270
const CARD_WIDTH = 270

const EmbeddingCard = ({ url, modelData }) => {
  const [hover, setHover] = useState(false)
  const [selected, setSelected] = useState(false)
  const { isCallingApi, setIsCallingApi } = useContext(ApiContext)
  const { isUpdatingModel } = useContext(ApiContext)
  const { setErrorMsg } = useContext(ApiContext)

  // UseEffects for parameter selection, change options based on previous selections
  useEffect(() => {}, [modelData])

  const launchModel = (url) => {
    if (isCallingApi || isUpdatingModel) {
      return
    }

    setIsCallingApi(true)

    const uuid = uuidv1()
    const modelDataWithID = {
      model_uid: uuid,
      model_name: modelData.model_name,
      model_type: 'embedding',
    }

    // First fetch request to initiate the model
    fetch(url + '/v1/models', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(modelDataWithID),
    })
      .then((res) => {
        if (!res.ok) {
          res
            .json()
            .then((errData) =>
              setErrorMsg(
                `Server error: ${res.status} - ${
                  errData.detail || 'Unknown error'
                }`
              )
            )
        } else {
          window.open(url + '/ui/#/running_models', '_blank', 'noreferrer')
        }
        setIsCallingApi(false)
      })
      .catch((error) => {
        console.error('Error:', error)
        setIsCallingApi(false)
      })
  }

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
    img: {
      display: 'block',
      margin: '0 auto',
      width: '180px',
      height: '180px',
      objectFit: 'cover',
      borderRadius: '10px',
    },
    titleContainer: {
      minHeight: '120px',
    },
    h2: {
      margin: '10px 10px',
      fontSize: '20px',
    },
    buttonsContainer: {
      display: 'flex',
      margin: '0 auto',
      marginTop: '120px',
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
      margin: '10px 0',
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
    langRow: {
      margin: '2px 5px 40px 5px',
    },
  }

  // Set two different states based on mouse hover
  return (
    <Box
      style={hover ? styles.containerSelected : styles.container}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={() => {
        if (!selected) {
          setSelected(true)
        }
      }}
    >
      {/* First state: show description page */}
      <Box style={styles.descriptionCard}>
        <div style={styles.titleContainer}>
          <h2 style={styles.h2}>{modelData.model_name}</h2>
          <div style={styles.langRow}>
            {(() => {
              if (modelData.language.includes('en')) {
                return (
                  <Chip
                    label="EN"
                    variant="outlined"
                    size="small"
                    sx={{ marginRight: '10px' }}
                  />
                )
              }
            })()}
            {(() => {
              if (modelData.language.includes('zh')) {
                return <Chip label="ZH" variant="outlined" size="small" />
              }
            })()}
          </div>
        </div>
        <div style={styles.iconRow}>
          <div style={styles.iconItem}>
            <span style={styles.boldIconText}>{modelData.dimensions}</span>
            <small style={styles.smallText}>dimensions</small>
          </div>
          <div style={styles.iconItem}>
            <span style={styles.boldIconText}>{modelData.max_tokens}</span>
            <small style={styles.smallText}>max tokens</small>
          </div>
        </div>
        {hover ? (
          <p style={styles.instructionText}>
            Click with mouse to launch the model
          </p>
        ) : (
          <p style={styles.instructionText}></p>
        )}
      </Box>
      {/* Second state: show parameter selection page */}
      <Box
        style={
          selected
            ? { ...styles.parameterCard, ...styles.slideIn }
            : { ...styles.parameterCard, ...styles.slideOut }
        }
      >
        <h2 style={styles.h2}>{modelData.model_name}</h2>
        <Box style={styles.buttonsContainer}>
          <button
            title="Launch Embedding"
            style={styles.buttonContainer}
            onClick={() => launchModel(url, modelData)}
            disabled={isCallingApi || isUpdatingModel || !modelData}
          >
            {(() => {
              if (isCallingApi || isUpdatingModel) {
                return (
                  <Box
                    style={{ ...styles.buttonItem, backgroundColor: '#f2f2f2' }}
                  >
                    <CircularProgress
                      size="20px"
                      sx={{
                        color: '#000000',
                      }}
                    />
                  </Box>
                )
              } else if (!modelData) {
                return (
                  <Box
                    style={{ ...styles.buttonItem, backgroundColor: '#f2f2f2' }}
                  >
                    <RocketLaunchOutlined size="20px" />
                  </Box>
                )
              } else {
                return (
                  <Box style={styles.buttonItem}>
                    <RocketLaunchOutlined color="#000000" size="20px" />
                  </Box>
                )
              }
            })()}
          </button>
          <button
            title="Launch Embedding"
            style={styles.buttonContainer}
            onClick={() => setSelected(false)}
          >
            <Box style={styles.buttonItem}>
              <UndoOutlined color="#000000" size="20px" />
            </Box>
          </button>
        </Box>
      </Box>
    </Box>
  )
}

export default EmbeddingCard
