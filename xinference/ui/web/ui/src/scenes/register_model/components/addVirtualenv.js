import AddIcon from '@mui/icons-material/Add'
import DeleteIcon from '@mui/icons-material/Delete'
import { Button, TextField } from '@mui/material'
import React, { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

const AddVirtualenv = ({ virtualenv, onChangeVirtualenv, scrollRef }) => {
  const { t } = useTranslation()

  const [packageKeys, setPackageKeys] = useState(() =>
    virtualenv.packages.map(() => `${Date.now()}-${Math.random()}`)
  )

  useEffect(() => {
    handleScrollBottom()
    setPackageKeys((prevKeys) => {
      const lengthDiff = virtualenv.packages.length - prevKeys.length

      if (lengthDiff > 0) {
        return [
          ...prevKeys,
          ...new Array(lengthDiff)
            .fill(0)
            .map(() => `${Date.now()}-${Math.random()}`),
        ]
      } else if (lengthDiff < 0) {
        return prevKeys.slice(0, virtualenv.packages.length)
      }

      return prevKeys
    })
  }, [virtualenv.packages])

  const handleScrollBottom = () => {
    scrollRef.current.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: 'smooth',
    })
  }

  return (
    <div>
      <label style={{ marginBottom: '20px', marginRight: '20px' }}>
        {t('registerModel.packages')}
      </label>
      <Button
        variant="contained"
        size="small"
        endIcon={<AddIcon />}
        onClick={() => onChangeVirtualenv('add')}
      >
        {t('registerModel.more')}
      </Button>

      <div>
        {virtualenv.packages.map((item, index) => (
          <div
            key={packageKeys[index]}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              marginTop: '10px',
              marginLeft: 50,
            }}
          >
            <TextField
              style={{ width: '100%' }}
              size="small"
              value={item}
              onChange={(e) => {
                onChangeVirtualenv('change', index, e.target.value)
              }}
            />
            <DeleteIcon
              onClick={() => onChangeVirtualenv('delete', index)}
              style={{ cursor: 'pointer', color: '#1976d2' }}
            />
          </div>
        ))}
      </div>
    </div>
  )
}

export default AddVirtualenv
